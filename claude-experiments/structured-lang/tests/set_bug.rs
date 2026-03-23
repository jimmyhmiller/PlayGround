//! Test per-row value tracking: Set edits are tracked per-row via
//! DiffChannel::RowData, so edits on different rows don't conflict
//! and edits on the same row DO merge correctly.

use structured_lang::database::Database;
use structured_lang::types::*;

#[test]
fn set_on_different_rows_should_not_conflict() {
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

    let bob = db.insert_row(main, "users", vec![
        ("name", Value::Str("Bob".into())),
        ("age", Value::Num(25.0)),
    ]).unwrap();

    let dev = db.fork_branch(main, "dev").unwrap();
    let staging = db.fork_branch(main, "staging").unwrap();
    db.diff_branches(dev, staging).unwrap();

    // Dev sets Alice's age to 31 (tracked in RowData(alice))
    db.set_field(dev, "users", alice, "age", Value::Num(31.0)).unwrap();

    // Staging sets Bob's age to 26 (tracked in RowData(bob))
    db.set_field(staging, "users", bob, "age", Value::Num(26.0)).unwrap();

    // Merge dev → staging
    db.merge_all(dev, staging).unwrap();

    // Bob's age should be 26 on staging — dev edited a DIFFERENT row, no conflict
    let bob_view = db.get_row(staging, "users", bob).unwrap();
    let bob_fields: std::collections::HashMap<String, serde_json::Value> =
        bob_view.fields.into_iter().collect();
    assert_eq!(bob_fields.get("age"), Some(&serde_json::json!(26.0)),
        "Bob's age should be 26 — dev's edit was on Alice, not Bob");

    // Alice's age SHOULD be 31 on staging — dev's Set on Alice migrated via RowData
    let alice_view = db.get_row(staging, "users", alice).unwrap();
    let alice_fields: std::collections::HashMap<String, serde_json::Value> =
        alice_view.fields.into_iter().collect();
    assert_eq!(alice_fields.get("age"), Some(&serde_json::json!(31.0)),
        "Alice's age should be 31 on staging — dev's per-row Set migrated");
}

#[test]
fn set_on_same_row_same_field_conflicts() {
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
    let staging = db.fork_branch(main, "staging").unwrap();
    db.diff_branches(dev, staging).unwrap();

    // Both edit Alice's age — SAME row, SAME field → conflict
    db.set_field(dev, "users", alice, "age", Value::Num(31.0)).unwrap();
    db.set_field(staging, "users", alice, "age", Value::Num(32.0)).unwrap();

    // Merge dev → staging: dev's Set wins (pre overrides diff in project)
    db.merge_all(dev, staging).unwrap();

    let alice_view = db.get_row(staging, "users", alice).unwrap();
    let alice_fields: std::collections::HashMap<String, serde_json::Value> =
        alice_view.fields.into_iter().collect();
    assert_eq!(alice_fields.get("age"), Some(&serde_json::json!(31.0)),
        "dev's Set should win the conflict (first migrated wins)");
}

#[test]
fn schema_and_value_edits_merge_together() {
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
    let staging = db.fork_branch(main, "staging").unwrap();
    db.diff_branches(dev, staging).unwrap();

    // Dev: set Alice's age to 31 (value edit)
    db.set_field(dev, "users", alice, "age", Value::Num(31.0)).unwrap();

    // Dev: add email column (schema edit)
    db.add_column(dev, "users", "email", AtomicType::Str).unwrap();

    // Merge dev → staging
    db.merge_all(dev, staging).unwrap();

    // Staging should have the email column (schema edit merged)
    let staging_schema = db.get_table_view(staging, "users").unwrap();
    let cols: Vec<&str> = staging_schema.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(cols.contains(&"email"), "schema edit should merge: {:?}", cols);

    // AND Alice's age should be 31 (per-row value edit also merged)
    let alice_view = db.get_row(staging, "users", alice).unwrap();
    let fields: std::collections::HashMap<String, serde_json::Value> =
        alice_view.fields.into_iter().collect();
    assert_eq!(fields.get("age"), Some(&serde_json::json!(31.0)),
        "value change should also migrate alongside schema change");
    assert_eq!(fields.get("email"), Some(&serde_json::json!("")),
        "new email column should have default value");
}
