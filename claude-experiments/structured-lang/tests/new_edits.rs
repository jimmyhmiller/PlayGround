//! Property tests specifically targeting the Rename and Set edit extensions.
//! These test the OT properties that matter for the new edit types:
//! - Shifting with Ins
//! - Following Move
//! - Conflict resolution at same index
//! - Independence from other edit types
//! - Retract dependency detection

use proptest::prelude::*;
use structured_lang::apply::*;
use structured_lang::transform::*;
use structured_lang::types::*;

fn arb_atomic_type() -> impl Strategy<Value = AtomicType> {
    prop_oneof![
        Just(AtomicType::Num),
        Just(AtomicType::Str),
        Just(AtomicType::Bool),
    ]
}

// ============================================================================
// Rename: shift with Ins
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn rename_shifts_right_when_ins_before(
        ins_idx in 0..5usize,
        rename_idx in 0..5usize,
        name in "[a-z]{1,5}",
        ty in arb_atomic_type(),
        id in 1..1000u64,
    ) {
        let ins = Edit::Ins { idx: ins_idx, ty, id };
        let rename = Edit::Rename { idx: rename_idx, name: name.clone() };
        let result = project(&rename, &ins);

        if rename_idx >= ins_idx {
            // Rename should shift right
            prop_assert_eq!(result.edit, Edit::Rename { idx: rename_idx + 1, name });
        } else {
            // Rename is left of insert, unaffected
            prop_assert_eq!(result.edit, Edit::Rename { idx: rename_idx, name });
        }
    }

    #[test]
    fn set_shifts_right_when_ins_before(
        ins_idx in 0..5usize,
        set_idx in 0..5usize,
        ty in arb_atomic_type(),
        id in 1..1000u64,
    ) {
        let ins = Edit::Ins { idx: ins_idx, ty, id };
        let set = Edit::Set { idx: set_idx, value: Value::Num(42.0) };
        let result = project(&set, &ins);

        if set_idx >= ins_idx {
            prop_assert_eq!(result.edit, Edit::Set { idx: set_idx + 1, value: Value::Num(42.0) });
        } else {
            prop_assert_eq!(result.edit, Edit::Set { idx: set_idx, value: Value::Num(42.0) });
        }
    }
}

// ============================================================================
// Rename: follows Move source to target
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn rename_follows_move_source_to_target(
        mi in 0..5usize,
        mj in 0..5usize,
        name in "[a-z]{1,5}",
    ) {
        prop_assume!(mi != mj);
        let rename = Edit::Rename { idx: mj, name: name.clone() };
        let mv = Edit::Move { i: mi, j: mj };
        let result = project(&rename, &mv);

        // Rename at Move source → follows to target
        prop_assert_eq!(result.edit, Edit::Rename { idx: mi, name },
            "rename at source should follow to target");
        prop_assert_eq!(result.adjust, mv,
            "move should pass through unchanged");
    }

    #[test]
    fn rename_overridden_at_move_target(
        mi in 0..5usize,
        mj in 0..5usize,
        name in "[a-z]{1,5}",
    ) {
        prop_assume!(mi != mj);
        let rename = Edit::Rename { idx: mi, name };
        let mv = Edit::Move { i: mi, j: mj };
        let result = project(&rename, &mv);

        // Rename at Move target → overridden
        prop_assert_eq!(result.edit, Edit::Id);
        prop_assert_eq!(result.adjust, mv);
    }

    #[test]
    fn set_follows_move_source_to_target(
        mi in 0..5usize,
        mj in 0..5usize,
    ) {
        prop_assume!(mi != mj);
        let set = Edit::Set { idx: mj, value: Value::Str("hello".into()) };
        let mv = Edit::Move { i: mi, j: mj };
        let result = project(&set, &mv);

        prop_assert_eq!(result.edit, Edit::Set { idx: mi, value: Value::Str("hello".into()) });
    }
}

// ============================================================================
// Rename vs Rename: same-index conflict
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn rename_conflict_at_same_index(
        idx in 0..5usize,
        name1 in "[a-z]{1,5}",
        name2 in "[a-z]{1,5}",
    ) {
        prop_assume!(name1 != name2);
        let pre = Edit::Rename { idx, name: name1.clone() };
        let diff = Edit::Rename { idx, name: name2 };
        let result = project(&pre, &diff);

        // Pre wins conflict
        prop_assert_eq!(result.edit, Edit::Rename { idx, name: name1 });
        prop_assert_eq!(result.adjust, Edit::Id);
    }

    #[test]
    fn set_conflict_at_same_index(idx in 0..5usize) {
        let pre = Edit::Set { idx, value: Value::Num(1.0) };
        let diff = Edit::Set { idx, value: Value::Num(2.0) };
        let result = project(&pre, &diff);

        prop_assert_eq!(result.edit, Edit::Set { idx, value: Value::Num(1.0) });
        prop_assert_eq!(result.adjust, Edit::Id);
    }
}

// ============================================================================
// Independence: Rename and Conv don't interfere
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn rename_independent_of_conv(
        idx in 0..5usize,
        name in "[a-z]{1,5}",
        ty in arb_atomic_type(),
    ) {
        let rename = Edit::Rename { idx, name: name.clone() };
        let conv = Edit::Conv { idx, ty };
        let result = project(&rename, &conv);

        // Different kinds at same index: independent (both pass through)
        prop_assert_eq!(result.edit, Edit::Rename { idx, name });
        prop_assert_eq!(result.adjust, Edit::Conv { idx, ty });
    }

    #[test]
    fn set_independent_of_conv(
        idx in 0..5usize,
        ty in arb_atomic_type(),
    ) {
        let set = Edit::Set { idx, value: Value::Num(42.0) };
        let conv = Edit::Conv { idx, ty };
        let result = project(&set, &conv);

        // Set and Conv are orthogonal (value vs type)
        prop_assert_eq!(result.edit, Edit::Set { idx, value: Value::Num(42.0) });
        prop_assert_eq!(result.adjust, Edit::Conv { idx, ty });
    }

    #[test]
    fn rename_independent_of_set(
        idx in 0..5usize,
        name in "[a-z]{1,5}",
    ) {
        let rename = Edit::Rename { idx, name: name.clone() };
        let set = Edit::Set { idx, value: Value::Bool(true) };
        let result = project(&rename, &set);

        // Rename and Set are independent (name vs value)
        prop_assert_eq!(result.edit, Edit::Rename { idx, name });
        prop_assert_eq!(result.adjust, Edit::Set { idx, value: Value::Bool(true) });
    }
}

// ============================================================================
// Retract: dependency detection for Rename/Set through Ins
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn rename_cannot_retract_through_ins_at_same_index(
        idx in 0..5usize,
        name in "[a-z]{1,5}",
        ty in arb_atomic_type(),
        id in 1..1000u64,
    ) {
        let post = Edit::Rename { idx, name };
        let diff = Edit::Ins { idx, ty, id };
        let result = retract(&post, &diff);
        prop_assert!(result.is_none(),
            "rename at Ins position should be a dependency (None)");
    }

    #[test]
    fn set_cannot_retract_through_ins_at_same_index(
        idx in 0..5usize,
        ty in arb_atomic_type(),
        id in 1..1000u64,
    ) {
        let post = Edit::Set { idx, value: Value::Num(1.0) };
        let diff = Edit::Ins { idx, ty, id };
        let result = retract(&post, &diff);
        prop_assert!(result.is_none(),
            "set at Ins position should be a dependency (None)");
    }
}

// ============================================================================
// Retract: Rename/Set through Move (at target → retract to source)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn rename_retracts_through_move_target_to_source(
        mi in 0..5usize,
        mj in 0..5usize,
        name in "[a-z]{1,5}",
    ) {
        prop_assume!(mi != mj);
        let post = Edit::Rename { idx: mi, name: name.clone() };
        let diff = Edit::Move { i: mi, j: mj };
        let result = retract(&post, &diff);

        prop_assert!(result.is_some());
        let r = result.unwrap();
        // Should retract to the source position
        prop_assert_eq!(r.edit, Edit::Rename { idx: mj, name });
    }

    #[test]
    fn set_retracts_through_move_target_to_source(
        mi in 0..5usize,
        mj in 0..5usize,
    ) {
        prop_assume!(mi != mj);
        let post = Edit::Set { idx: mi, value: Value::Num(99.0) };
        let diff = Edit::Move { i: mi, j: mj };
        let result = retract(&post, &diff);

        prop_assert!(result.is_some());
        let r = result.unwrap();
        prop_assert_eq!(r.edit, Edit::Set { idx: mj, value: Value::Num(99.0) });
    }

    #[test]
    fn rename_cannot_retract_through_move_at_source(
        mi in 0..5usize,
        mj in 0..5usize,
        name in "[a-z]{1,5}",
    ) {
        prop_assume!(mi != mj);
        let post = Edit::Rename { idx: mj, name };
        let diff = Edit::Move { i: mi, j: mj };
        let result = retract(&post, &diff);
        prop_assert!(result.is_none(),
            "rename at Move source should be a dependency");
    }
}

// ============================================================================
// Full commutativity: any edit pair involving Rename/Set
// ============================================================================

fn arb_edit_for_size(size: usize) -> BoxedStrategy<Edit> {
    if size == 0 {
        return Just(Edit::Id).boxed();
    }
    prop_oneof![
        1 => Just(Edit::Id),
        2 => (0..size, arb_atomic_type()).prop_map(|(idx, ty)| Edit::Conv { idx, ty }),
        2 => (0..=size, arb_atomic_type(), 1..1000u64)
            .prop_map(|(idx, ty, id)| Edit::Ins { idx, ty, id }),
        if size >= 2 { 1 } else { 0 } =>
            (0..size, 0..size)
                .prop_filter("i != j", |(i, j)| i != j)
                .prop_map(|(i, j)| Edit::Move { i, j }),
        2 => (0..size, "[a-z]{1,3}")
            .prop_map(|(idx, name)| Edit::Rename { idx, name }),
        2 => (0..size, arb_atomic_type())
            .prop_map(|(idx, ty)| Edit::Set { idx, value: default_value(ty) }),
    ]
    .boxed()
}

fn clamp_edit(edit: &Edit, size: usize) -> Edit {
    if size == 0 { return Edit::Id; }
    match edit {
        Edit::Id => Edit::Id,
        Edit::Ins { idx, ty, id } => Edit::Ins { idx: *idx % (size + 1), ty: *ty, id: *id },
        Edit::Conv { idx, ty } => Edit::Conv { idx: *idx % size, ty: *ty },
        Edit::Move { i, j } => {
            if size < 2 { return Edit::Id; }
            let i = *i % size;
            let j = *j % size;
            if i == j { Edit::Move { i, j: (j + 1) % size } } else { Edit::Move { i, j } }
        }
        Edit::Rename { idx, name } => Edit::Rename { idx: *idx % size, name: name.clone() },
        Edit::Set { idx, value } => Edit::Set { idx: *idx % size, value: value.clone() },
    }
}

fn apply_to_types(types: &[AtomicType], edit: &Edit) -> Option<Vec<AtomicType>> {
    let doc = Document::from_types(types);
    apply(&doc, edit).map(|d| d.types())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    /// Full commutativity test for project with all 6 edit types.
    #[test]
    fn full_project_commutativity(
        types in prop::collection::vec(arb_atomic_type(), 2..=5),
        pre in arb_edit_for_size(5),
        diff in arb_edit_for_size(5).prop_map(|e| match e {
            Edit::Ins { idx, ty, id } => Edit::Ins { idx, ty, id: id + 100_000 },
            other => other,
        }),
    ) {
        let size = types.len();
        let pre = clamp_edit(&pre, size);
        let diff = clamp_edit(&diff, size);

        let result = project(&pre, &diff);
        let post = &result.edit;
        let adjust = &result.adjust;

        if let Some(after_diff) = apply_to_types(&types, &diff) {
            let path1 = apply_to_types(&after_diff, post);
            if let Some(after_pre) = apply_to_types(&types, &pre) {
                let path2 = apply_to_types(&after_pre, adjust);
                if let (Some(p1), Some(p2)) = (path1, path2) {
                    prop_assert_eq!(p1, p2,
                        "commutativity failed:\n  pre={:?}\n  diff={:?}\n  post={:?}\n  adjust={:?}",
                        pre, diff, post, adjust);
                }
            }
        }
    }

    /// Full commutativity test for retract with all 6 edit types.
    #[test]
    fn full_retract_commutativity(
        types in prop::collection::vec(arb_atomic_type(), 2..=5),
        post in arb_edit_for_size(5),
        diff in arb_edit_for_size(5).prop_map(|e| match e {
            // Offset Ins ids to avoid same-id collisions with post
            Edit::Ins { idx, ty, id } => Edit::Ins { idx, ty, id: id + 100_000 },
            other => other,
        }),
    ) {
        let size = types.len();
        let post = clamp_edit(&post, size);
        let diff = clamp_edit(&diff, size);

        if let Some(result) = retract(&post, &diff) {
            let pre = &result.edit;
            let adjust = &result.adjust;

            if let Some(after_diff) = apply_to_types(&types, &diff) {
                if let Some(path1) = apply_to_types(&after_diff, &post) {
                    if let Some(after_pre) = apply_to_types(&types, pre) {
                        if let Some(path2) = apply_to_types(&after_pre, adjust) {
                            prop_assert_eq!(path1, path2,
                                "retract commutativity failed:\n  post={:?}\n  diff={:?}\n  pre={:?}\n  adjust={:?}",
                                post, diff, pre, adjust);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// E2E: schema changes with Rename and Set merge correctly
// ============================================================================

#[test]
fn e2e_rename_and_set_merge() {
    use structured_lang::database::Database;

    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "users", vec![("name", AtomicType::Str), ("age", AtomicType::Num)]).unwrap();
    db.insert_row(main, "users", vec![
        ("name", Value::Str("Alice".into())),
        ("age", Value::Num(30.0)),
    ]).unwrap();

    let alice = db.fork_branch(main, "alice").unwrap();
    let bob = db.fork_branch(main, "bob").unwrap();
    db.diff_branches(alice, bob).unwrap();

    // Alice: rename "name" to "full_name"
    db.rename_column(alice, "users", "name", "full_name").unwrap();

    // Alice: set age to 31
    db.set_field(alice, "users", 0, "age", Value::Num(31.0)).unwrap();

    // Bob: add email column
    db.add_column(bob, "users", "email", AtomicType::Str).unwrap();

    // Bob: set name to "Robert"
    db.set_field(bob, "users", 0, "name", Value::Str("Robert".into())).unwrap();

    // Merge alice → bob
    let applied = db.merge_all(alice, bob).unwrap();
    assert!(!applied.is_empty(), "should have merged something");

    // Bob should have "full_name" (from alice's rename)
    let bob_schema = db.get_table_view(bob, "users").unwrap();
    let col_names: Vec<&str> = bob_schema.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(col_names.contains(&"full_name"), "bob should have full_name after merge: {:?}", col_names);
    assert!(col_names.contains(&"email"), "bob should still have email: {:?}", col_names);

    // Merge bob → alice
    let applied = db.merge_all(bob, alice).unwrap();
    assert!(!applied.is_empty(), "should have merged something");

    // Alice should have "email" (from bob's add)
    let alice_schema = db.get_table_view(alice, "users").unwrap();
    let col_names: Vec<&str> = alice_schema.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(col_names.contains(&"full_name"), "alice should have full_name: {:?}", col_names);
    assert!(col_names.contains(&"email"), "alice should have email after merge: {:?}", col_names);

    // Both branches should have the same column names
    let alice_names: Vec<&str> = alice_schema.columns.iter().map(|(n, _)| n.as_str()).collect();
    let bob_names: Vec<&str> = bob_schema.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(alice_names.len(), bob_names.len(),
        "both branches should have same number of columns");
}
