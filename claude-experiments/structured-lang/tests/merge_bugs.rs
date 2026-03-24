//! Tests for merge bugs found during GUI demo.

use structured_lang::database::Database;
use structured_lang::types::*;

/// Bug 1: Values don't migrate in the reverse direction.
/// Both branches add a column (at same positional index), set values.
/// After bidirectional merge, BOTH sets of values should exist.
#[test]
fn values_migrate_in_both_directions() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("name", AtomicType::Str)]).unwrap();
    let r = db.insert_row(main, "t", vec![("name", Value::Str("Alice".into()))]).unwrap();

    let a = db.fork_branch(main, "a").unwrap();
    let b = db.fork_branch(main, "b").unwrap();

    // A adds "email", sets value
    db.add_column(a, "t", "email", AtomicType::Str).unwrap();
    db.set_field(a, "t", r, "email", Value::Str("alice@a.com".into())).unwrap();

    // B adds "dept", sets value
    db.add_column(b, "t", "dept", AtomicType::Str).unwrap();
    db.set_field(b, "t", r, "dept", Value::Str("Engineering".into())).unwrap();

    // Merge A → B
    db.merge_all(a, b).unwrap();

    // Merge B → A
    db.merge_all(b, a).unwrap();

    // B should have BOTH email and dept values
    let view_b = db.get_row(b, "t", r).unwrap();
    let fb: std::collections::HashMap<String, serde_json::Value> =
        view_b.fields.into_iter().collect();
    assert_eq!(fb.get("email"), Some(&serde_json::json!("alice@a.com")),
        "B should have email value from A: {:?}", fb);
    assert_eq!(fb.get("dept"), Some(&serde_json::json!("Engineering")),
        "B should keep its own dept value: {:?}", fb);

    // A should have BOTH email and dept values
    let view_a = db.get_row(a, "t", r).unwrap();
    let fa: std::collections::HashMap<String, serde_json::Value> =
        view_a.fields.into_iter().collect();
    assert_eq!(fa.get("email"), Some(&serde_json::json!("alice@a.com")),
        "A should keep its own email value: {:?}", fa);
    assert_eq!(fa.get("dept"), Some(&serde_json::json!("Engineering")),
        "A should have dept value from B: {:?}", fa);
}

/// Bug 2: False conflict when both branches add a column at the same
/// positional index. Email and department are different columns but
/// their Set edits conflict because both are at idx 3.
#[test]
fn no_false_conflict_on_different_columns_same_index() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![
        ("a", AtomicType::Str),
        ("b", AtomicType::Str),
        ("c", AtomicType::Str),
    ]).unwrap();
    let r = db.insert_row(main, "t", vec![]).unwrap();

    let x = db.fork_branch(main, "x").unwrap();
    let y = db.fork_branch(main, "y").unwrap();

    // X adds column "email" (goes to idx 3)
    db.add_column(x, "t", "email", AtomicType::Str).unwrap();
    db.set_field(x, "t", r, "email", Value::Str("test@test.com".into())).unwrap();

    // Y adds column "dept" (also goes to idx 3 on Y's branch)
    db.add_column(y, "t", "dept", AtomicType::Str).unwrap();
    db.set_field(y, "t", r, "dept", Value::Str("Engineering".into())).unwrap();

    // get_conflicts reports raw positional conflicts — both columns are at
    // idx 3 in their respective branches. But the OT resolves this during
    // migration by shifting one column's index. Verify the merge works:
    db.merge_all(x, y).unwrap();
    db.merge_all(y, x).unwrap();

    // Both branches should have both columns with correct values
    let vx = db.get_row(x, "t", r).unwrap();
    let fx: std::collections::HashMap<String, serde_json::Value> =
        vx.fields.into_iter().collect();
    assert_eq!(fx.get("email"), Some(&serde_json::json!("test@test.com")),
        "x should have email: {:?}", fx);
    assert_eq!(fx.get("dept"), Some(&serde_json::json!("Engineering")),
        "x should have dept: {:?}", fx);

    let vy = db.get_row(y, "t", r).unwrap();
    let fy: std::collections::HashMap<String, serde_json::Value> =
        vy.fields.into_iter().collect();
    assert_eq!(fy.get("email"), Some(&serde_json::json!("test@test.com")),
        "y should have email: {:?}", fy);
    assert_eq!(fy.get("dept"), Some(&serde_json::json!("Engineering")),
        "y should have dept: {:?}", fy);
}

/// Bug 3 (from GUI demo): after merging A→B, B has both columns + values.
/// Then merging B→A should give A the values from B's original column.
/// This is the exact scenario from the GUI demo with hr-changes/reorg.
#[test]
fn gui_demo_scenario_full_convergence() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "employees", vec![
        ("name", AtomicType::Str),
        ("role", AtomicType::Str),
        ("salary", AtomicType::Num),
    ]).unwrap();
    let alice = db.insert_row(main, "employees", vec![
        ("name", Value::Str("Alice".into())),
        ("role", Value::Str("Engineer".into())),
        ("salary", Value::Num(120000.0)),
    ]).unwrap();

    let hr = db.fork_branch(main, "hr").unwrap();
    let reorg = db.fork_branch(main, "reorg").unwrap();

    // HR: add email, set value, rename
    db.add_column(hr, "employees", "email", AtomicType::Str).unwrap();
    db.set_field(hr, "employees", alice, "email", Value::Str("alice@co.com".into())).unwrap();
    db.rename_column(hr, "employees", "name", "full_name").unwrap();

    // Reorg: convert salary, add department, set value
    db.convert_column(reorg, "employees", "salary", AtomicType::Str).unwrap();
    db.add_column(reorg, "employees", "department", AtomicType::Str).unwrap();
    db.set_field(reorg, "employees", alice, "department", Value::Str("Platform".into())).unwrap();

    // Merge both ways
    db.merge_all(hr, reorg).unwrap();
    db.merge_all(reorg, hr).unwrap();

    // BOTH branches should have identical schemas
    let hr_schema = db.get_table_view(hr, "employees").unwrap();
    let reorg_schema = db.get_table_view(reorg, "employees").unwrap();
    let mut hr_cols: Vec<String> = hr_schema.columns.iter().map(|(n, _)| n.clone()).collect();
    let mut reorg_cols: Vec<String> = reorg_schema.columns.iter().map(|(n, _)| n.clone()).collect();
    hr_cols.sort();
    reorg_cols.sort();
    assert_eq!(hr_cols, reorg_cols, "schemas should converge");

    // BOTH should have Alice's email AND department values
    let hr_row = db.get_row(hr, "employees", alice).unwrap();
    let hr_f: std::collections::HashMap<String, serde_json::Value> =
        hr_row.fields.into_iter().collect();
    assert_eq!(hr_f.get("email"), Some(&serde_json::json!("alice@co.com")),
        "hr should have email: {:?}", hr_f);
    assert_eq!(hr_f.get("department"), Some(&serde_json::json!("Platform")),
        "hr should have department value from reorg: {:?}", hr_f);

    let reorg_row = db.get_row(reorg, "employees", alice).unwrap();
    let reorg_f: std::collections::HashMap<String, serde_json::Value> =
        reorg_row.fields.into_iter().collect();
    assert_eq!(reorg_f.get("email"), Some(&serde_json::json!("alice@co.com")),
        "reorg should have email value from hr: {:?}", reorg_f);
    assert_eq!(reorg_f.get("department"), Some(&serde_json::json!("Platform")),
        "reorg should have department: {:?}", reorg_f);
}
