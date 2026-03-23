//! Comprehensive end-to-end test: two users, multiple tables, all edit types,
//! full merge in both directions, verify convergence.

use structured_lang::database::Database;
use structured_lang::types::*;

#[test]
fn full_collaboration_scenario() {
    let mut db = Database::new();

    // ── Setup: main branch with two tables ─────────────────────────────
    let main = db.create_branch("main");

    db.create_table(main, "users", vec![
        ("name", AtomicType::Str),
        ("age", AtomicType::Num),
    ]).unwrap();

    db.create_table(main, "posts", vec![
        ("title", AtomicType::Str),
        ("body", AtomicType::Str),
    ]).unwrap();

    // Insert data
    let alice_row = db.insert_row(main, "users", vec![
        ("name", Value::Str("Alice".into())),
        ("age", Value::Num(30.0)),
    ]).unwrap();

    let bob_row = db.insert_row(main, "users", vec![
        ("name", Value::Str("Bob".into())),
        ("age", Value::Num(25.0)),
    ]).unwrap();

    db.insert_row(main, "posts", vec![
        ("title", Value::Str("Hello World".into())),
        ("body", Value::Str("My first post".into())),
    ]).unwrap();

    // ── Fork into two developer branches ───────────────────────────────
    let dev = db.fork_branch(main, "dev").unwrap();
    let staging = db.fork_branch(main, "staging").unwrap();

    // Start tracking diffs
    db.diff_branches(dev, staging).unwrap();

    // ── Dev: schema edits on users table ───────────────────────────────
    // Add email column
    db.add_column(dev, "users", "email", AtomicType::Str).unwrap();

    // Convert age from Num to Str
    db.convert_column(dev, "users", "age", AtomicType::Str).unwrap();

    // Rename "name" to "full_name"
    db.rename_column(dev, "users", "name", "full_name").unwrap();

    // ── Staging: schema edits on posts table ───────────────────────────
    // Add published flag
    db.add_column(staging, "posts", "published", AtomicType::Bool).unwrap();

    // Rename body to content
    db.rename_column(staging, "posts", "body", "content").unwrap();

    // Add category column to users
    db.add_column(staging, "users", "category", AtomicType::Str).unwrap();

    // ── Verify branches diverged ───────────────────────────────────────
    let dev_users = db.get_table_view(dev, "users").unwrap();
    let dev_cols: Vec<&str> = dev_users.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(dev_cols.contains(&"full_name"), "dev should have full_name: {:?}", dev_cols);
    assert!(dev_cols.contains(&"email"), "dev should have email: {:?}", dev_cols);
    assert!(!dev_cols.contains(&"category"), "dev should NOT have category yet: {:?}", dev_cols);

    let staging_posts = db.get_table_view(staging, "posts").unwrap();
    let staging_cols: Vec<&str> = staging_posts.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(staging_cols.contains(&"published"), "staging should have published: {:?}", staging_cols);
    assert!(staging_cols.contains(&"content"), "staging should have content: {:?}", staging_cols);

    let staging_users = db.get_table_view(staging, "users").unwrap();
    let staging_user_cols: Vec<&str> = staging_users.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(staging_user_cols.contains(&"category"), "staging should have category: {:?}", staging_user_cols);
    assert!(!staging_user_cols.contains(&"email"), "staging should NOT have email yet: {:?}", staging_user_cols);

    // ── Check diffs ────────────────────────────────────────────────────
    let diffs = db.get_diffs(dev, staging);
    assert!(!diffs.is_empty(), "should have diffs");
    // Both tables should show diffs
    let diff_tables: Vec<&str> = diffs.iter().map(|(t, _, _)| t.as_str()).collect();
    assert!(diff_tables.contains(&"users"), "users table should have diffs");
    assert!(diff_tables.contains(&"posts"), "posts table should have diffs");

    // ── Merge dev → staging ────────────────────────────────────────────
    let applied = db.merge_all(dev, staging).unwrap();
    assert!(!applied.is_empty(), "should have applied edits");

    // Staging should now have dev's users changes
    let staging_users = db.get_table_view(staging, "users").unwrap();
    let cols: Vec<&str> = staging_users.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(cols.contains(&"full_name"), "staging should have full_name after merge: {:?}", cols);
    assert!(cols.contains(&"email"), "staging should have email after merge: {:?}", cols);
    assert!(cols.contains(&"category"), "staging should still have category: {:?}", cols);

    // Verify age was converted to Str
    let age_type = staging_users.columns.iter()
        .find(|(n, _)| n == "age")
        .map(|(_, t)| t.as_str());
    assert_eq!(age_type, Some("Str"), "age should be Str after merge");

    // ── Merge staging → dev ────────────────────────────────────────────
    let applied = db.merge_all(staging, dev).unwrap();
    assert!(!applied.is_empty(), "should have applied edits from staging");

    // Dev should now have staging's posts changes
    let dev_posts = db.get_table_view(dev, "posts").unwrap();
    let cols: Vec<&str> = dev_posts.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(cols.contains(&"published"), "dev should have published after merge: {:?}", cols);
    assert!(cols.contains(&"content"), "dev should have content after merge: {:?}", cols);

    // Dev should also have category on users
    let dev_users = db.get_table_view(dev, "users").unwrap();
    let cols: Vec<&str> = dev_users.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(cols.contains(&"category"), "dev should have category after merge: {:?}", cols);

    // ── Verify convergence: both branches have identical schemas ───────
    let dev_users = db.get_table_view(dev, "users").unwrap();
    let staging_users = db.get_table_view(staging, "users").unwrap();
    let mut dev_user_cols: Vec<String> = dev_users.columns.iter().map(|(n, _)| n.clone()).collect();
    let mut staging_user_cols: Vec<String> = staging_users.columns.iter().map(|(n, _)| n.clone()).collect();
    dev_user_cols.sort();
    staging_user_cols.sort();
    assert_eq!(dev_user_cols, staging_user_cols,
        "users columns should be identical after full merge");

    let dev_posts = db.get_table_view(dev, "posts").unwrap();
    let staging_posts = db.get_table_view(staging, "posts").unwrap();
    let mut dev_post_cols: Vec<String> = dev_posts.columns.iter().map(|(n, _)| n.clone()).collect();
    let mut staging_post_cols: Vec<String> = staging_posts.columns.iter().map(|(n, _)| n.clone()).collect();
    dev_post_cols.sort();
    staging_post_cols.sort();
    assert_eq!(dev_post_cols, staging_post_cols,
        "posts columns should be identical after full merge");

    // ── Verify no remaining diffs ──────────────────────────────────────
    let diffs = db.get_diffs(dev, staging);
    for (table, dev_count, staging_count) in &diffs {
        assert_eq!(*dev_count, 0,
            "dev should have 0 remaining diffs on {}", table);
        assert_eq!(*staging_count, 0,
            "staging should have 0 remaining diffs on {}", table);
    }

    // ── Verify data survived the schema changes ────────────────────────
    let alice = db.get_row(dev, "users", alice_row).unwrap();
    let alice_fields: std::collections::HashMap<String, serde_json::Value> =
        alice.fields.into_iter().collect();

    assert_eq!(alice_fields.get("full_name"),
        Some(&serde_json::json!("Alice")),
        "Alice's name should be preserved through rename");

    // Age was converted from Num(30) to Str type — conform shows "30"
    assert_eq!(alice_fields.get("age"),
        Some(&serde_json::json!("30")),
        "Alice's age should be converted to string '30'");

    // Email should have default value (empty string for Str)
    assert_eq!(alice_fields.get("email"),
        Some(&serde_json::json!("")),
        "email should have Str default");

    // Category should have default value
    assert_eq!(alice_fields.get("category"),
        Some(&serde_json::json!("")),
        "category should have Str default");

    let bob = db.get_row(dev, "users", bob_row).unwrap();
    let bob_fields: std::collections::HashMap<String, serde_json::Value> =
        bob.fields.into_iter().collect();

    assert_eq!(bob_fields.get("full_name"),
        Some(&serde_json::json!("Bob")),
        "Bob's name should be preserved through rename");

    assert_eq!(bob_fields.get("age"),
        Some(&serde_json::json!("25")),
        "Bob's age should be converted to string '25'");

    // ── Save and reload ────────────────────────────────────────────────
    let tmp = std::env::temp_dir().join("structured_lang_test.json");
    db.save(&tmp).unwrap();
    let db2 = Database::load(&tmp).unwrap();

    // Verify loaded DB has the same data
    let alice2 = db2.get_row(dev, "users", alice_row).unwrap();
    let alice2_fields: std::collections::HashMap<String, serde_json::Value> =
        alice2.fields.into_iter().collect();
    assert_eq!(alice2_fields.get("full_name"),
        Some(&serde_json::json!("Alice")),
        "data should survive save/load");

    std::fs::remove_file(tmp).ok();
}
