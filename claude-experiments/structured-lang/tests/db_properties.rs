//! Property tests for the DATABASE layer — not just the OT algebra.
//! These test the full pipeline: fork → edit → merge → verify.
//! The OT algebra property tests (tests/properties.rs) verify commutativity
//! of project/retract. These tests verify that the database correctly uses
//! that algebra to produce correct merge results.

use proptest::prelude::*;
use structured_lang::database::Database;
use structured_lang::types::*;

// ============================================================================
// Invariant 1: Schema convergence after full bidirectional merge
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn schema_converges_after_full_merge(
        n_edits_a in 1..4usize,
        n_edits_b in 1..4usize,
        seed in 0..1000u64,
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("x", AtomicType::Str), ("y", AtomicType::Num)]).unwrap();
        db.insert_row(main, "t", vec![("x", Value::Str("a".into())), ("y", Value::Num(1.0))]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(main, "b").unwrap();
        db.diff_branches(a, b).unwrap();

        // Apply random schema edits to each branch
        for i in 0..n_edits_a {
            let name = format!("col_a_{}", seed * 100 + i as u64);
            let _ = db.add_column(a, "t", &name, AtomicType::Str);
        }
        for i in 0..n_edits_b {
            let name = format!("col_b_{}", seed * 100 + i as u64);
            let _ = db.add_column(b, "t", &name, AtomicType::Str);
        }

        // Merge both ways
        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, a);

        // Schemas should converge
        let sa = db.get_table_view(a, "t").unwrap();
        let sb = db.get_table_view(b, "t").unwrap();
        let mut cols_a: Vec<String> = sa.columns.iter().map(|(n, _)| n.clone()).collect();
        let mut cols_b: Vec<String> = sb.columns.iter().map(|(n, _)| n.clone()).collect();
        cols_a.sort();
        cols_b.sort();
        prop_assert_eq!(&cols_a, &cols_b,
            "schemas should converge after full bidirectional merge");
    }
}

// ============================================================================
// Invariant 2: Value edits on one branch arrive on the other after merge
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn value_edits_propagate_through_merge(
        val in "[a-z]{1,5}",
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("name", AtomicType::Str)]).unwrap();
        let row = db.insert_row(main, "t", vec![("name", Value::Str("init".into()))]).unwrap();

        let dev = db.fork_branch(main, "dev").unwrap();

        // Set value on dev
        db.set_field(dev, "t", row, "name", Value::Str(val.clone())).unwrap();

        // Merge dev → main
        let _ = db.merge_all(dev, main);

        // Main should have the value
        let view = db.get_row(main, "t", row).unwrap();
        let fields: std::collections::HashMap<String, serde_json::Value> =
            view.fields.into_iter().collect();
        prop_assert_eq!(
            fields.get("name"),
            Some(&serde_json::json!(val)),
            "value should propagate through merge"
        );
    }
}

// ============================================================================
// Invariant 3: Add column + set value always arrives together
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn add_column_and_set_value_arrive_together(
        col_name in "[a-z]{1,5}",
        val in "[a-z]{1,10}",
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("base", AtomicType::Str)]).unwrap();
        let row = db.insert_row(main, "t", vec![("base", Value::Str("x".into()))]).unwrap();

        let dev = db.fork_branch(main, "dev").unwrap();

        // Add column and set value on dev
        if db.add_column(dev, "t", &col_name, AtomicType::Str).is_ok() {
            db.set_field(dev, "t", row, &col_name, Value::Str(val.clone())).unwrap();

            // Merge dev → main
            let _ = db.merge_all(dev, main);

            // Main should have BOTH the column AND the value
            let schema = db.get_table_view(main, "t").unwrap();
            let cols: Vec<&str> = schema.columns.iter().map(|(n, _)| n.as_str()).collect();
            prop_assert!(cols.contains(&col_name.as_str()),
                "column should exist after merge");

            let view = db.get_row(main, "t", row).unwrap();
            let fields: std::collections::HashMap<String, serde_json::Value> =
                view.fields.into_iter().collect();
            prop_assert_eq!(
                fields.get(&col_name),
                Some(&serde_json::json!(val)),
                "value should arrive with the column, not be default"
            );
        }
    }
}

// ============================================================================
// Invariant 4: Value edits on different rows don't interfere
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn different_row_edits_dont_interfere(
        val_a in "[a-z]{1,5}",
        val_b in "[a-z]{1,5}",
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("v", AtomicType::Str)]).unwrap();
        let r1 = db.insert_row(main, "t", vec![("v", Value::Str("init1".into()))]).unwrap();
        let r2 = db.insert_row(main, "t", vec![("v", Value::Str("init2".into()))]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(main, "b").unwrap();
        db.diff_branches(a, b).unwrap();

        // A edits row 1, B edits row 2
        db.set_field(a, "t", r1, "v", Value::Str(val_a.clone())).unwrap();
        db.set_field(b, "t", r2, "v", Value::Str(val_b.clone())).unwrap();

        // Merge both ways
        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, a);

        // Both branches should have both values
        for branch in [a, b] {
            let v1 = db.get_row(branch, "t", r1).unwrap();
            let f1: std::collections::HashMap<String, serde_json::Value> =
                v1.fields.into_iter().collect();
            prop_assert_eq!(f1.get("v"), Some(&serde_json::json!(val_a)),
                "row 1 should have val_a on branch {}", branch);

            let v2 = db.get_row(branch, "t", r2).unwrap();
            let f2: std::collections::HashMap<String, serde_json::Value> =
                v2.fields.into_iter().collect();
            prop_assert_eq!(f2.get("v"), Some(&serde_json::json!(val_b)),
                "row 2 should have val_b on branch {}", branch);
        }
    }
}

// ============================================================================
// Invariant 5: Schema edit + value edit on different branches compose
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn schema_edit_and_value_edit_compose(
        col_name in "[a-z]{1,5}",
        val in "[a-z]{1,5}",
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("name", AtomicType::Str)]).unwrap();
        let row = db.insert_row(main, "t", vec![("name", Value::Str("Alice".into()))]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(main, "b").unwrap();
        db.diff_branches(a, b).unwrap();

        // A: add column
        if db.add_column(a, "t", &col_name, AtomicType::Str).is_ok() {
            // B: set existing field value
            db.set_field(b, "t", row, "name", Value::Str(val.clone())).unwrap();

            // Merge both ways
            let _ = db.merge_all(a, b);
            let _ = db.merge_all(b, a);

            // Both should have the new column AND the value change
            for branch in [a, b] {
                let schema = db.get_table_view(branch, "t").unwrap();
                let cols: Vec<&str> = schema.columns.iter().map(|(n, _)| n.as_str()).collect();
                prop_assert!(cols.contains(&col_name.as_str()),
                    "branch {} should have new column", branch);

                let view = db.get_row(branch, "t", row).unwrap();
                let fields: std::collections::HashMap<String, serde_json::Value> =
                    view.fields.into_iter().collect();
                prop_assert_eq!(fields.get("name"), Some(&serde_json::json!(val)),
                    "branch {} should have updated name value", branch);
            }
        }
    }
}

// ============================================================================
// Invariant 6: No edits are silently dropped (count check)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn no_edits_silently_dropped(
        n_cols in 1..3usize,
        n_sets in 1..3usize,
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("base", AtomicType::Str)]).unwrap();
        let row = db.insert_row(main, "t", vec![("base", Value::Str("x".into()))]).unwrap();

        let dev = db.fork_branch(main, "dev").unwrap();

        // Add columns on dev
        let mut added_cols = Vec::new();
        for i in 0..n_cols {
            let name = format!("c{}", i);
            if db.add_column(dev, "t", &name, AtomicType::Str).is_ok() {
                added_cols.push(name);
            }
        }

        // Set values on added columns
        for (i, col) in added_cols.iter().enumerate() {
            for j in 0..n_sets {
                let val = format!("v{}_{}", i, j);
                let _ = db.set_field(dev, "t", row, col, Value::Str(val));
            }
        }

        // Merge dev → main
        let _ = db.merge_all(dev, main);

        // Main should have ALL added columns
        let schema = db.get_table_view(main, "t").unwrap();
        let cols: Vec<&str> = schema.columns.iter().map(|(n, _)| n.as_str()).collect();
        for col in &added_cols {
            prop_assert!(cols.contains(&col.as_str()),
                "column {} should exist on main after merge", col);
        }

        // Main should have the LAST set value for each column (not default)
        let view = db.get_row(main, "t", row).unwrap();
        let fields: std::collections::HashMap<String, serde_json::Value> =
            view.fields.into_iter().collect();
        for (i, col) in added_cols.iter().enumerate() {
            let expected = format!("v{}_{}", i, n_sets - 1);
            prop_assert_eq!(fields.get(col), Some(&serde_json::json!(expected)),
                "column {} should have last set value, not default", col);
        }
    }
}
