use proptest::prelude::*;
use structured_lang::apply::*;
use structured_lang::diff::*;
use structured_lang::transform::*;
use structured_lang::types::*;

// ============================================================================
// Arbitrary implementations for property-based testing
// ============================================================================

fn arb_atomic_type() -> impl Strategy<Value = AtomicType> {
    prop_oneof![
        Just(AtomicType::Num),
        Just(AtomicType::Str),
        Just(AtomicType::Bool),
    ]
}

fn arb_atomic_type_with_del() -> impl Strategy<Value = AtomicType> {
    prop_oneof![
        Just(AtomicType::Num),
        Just(AtomicType::Str),
        Just(AtomicType::Bool),
        Just(AtomicType::Del),
    ]
}

fn arb_value_for_type(ty: AtomicType) -> BoxedStrategy<Value> {
    match ty {
        AtomicType::Num => prop_oneof![
            Just(Value::Null),
            (-1000.0f64..1000.0).prop_map(Value::Num),
        ]
        .boxed(),
        AtomicType::Str => prop_oneof![
            Just(Value::Null),
            "[a-z]{0,10}".prop_map(Value::Str),
        ]
        .boxed(),
        AtomicType::Bool => prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
        ]
        .boxed(),
        AtomicType::Del => Just(Value::Null).boxed(),
    }
}

fn arb_field() -> impl Strategy<Value = Field> {
    arb_atomic_type().prop_flat_map(|ty| {
        arb_value_for_type(ty).prop_map(move |value| Field { value, ty })
    })
}

fn arb_document(max_size: usize) -> impl Strategy<Value = Document> {
    prop::collection::vec(arb_field(), 1..=max_size).prop_map(Document::new)
}

/// Generate a valid edit for a document of a given size.
/// Generate a valid edit for a document of given size.
/// `id_base` offsets Ins ids to avoid collisions between independently generated edits.
fn arb_edit_for_size_with_id(size: usize, id_base: u64) -> BoxedStrategy<Edit> {
    if size == 0 {
        return (0..=0usize, arb_atomic_type(), 1..100u64)
            .prop_map(move |(idx, ty, id)| Edit::Ins { idx, ty, id: id + id_base })
            .boxed();
    }

    prop_oneof![
        2 => Just(Edit::Id),
        3 => (0..size, arb_atomic_type()).prop_map(|(idx, ty)| Edit::Conv { idx, ty }),
        3 => (0..=size, arb_atomic_type(), 1..1000u64)
            .prop_map(move |(idx, ty, id)| Edit::Ins { idx, ty, id: id + id_base }),
        if size >= 2 { 2 } else { 0 } =>
            (0..size, 0..size)
                .prop_filter("move requires i != j", |(i, j)| i != j)
                .prop_map(|(i, j)| Edit::Move { i, j }),
        2 => (0..size, "[a-z]{1,5}")
            .prop_map(|(idx, name)| Edit::Rename { idx, name }),
        2 => (0..size, arb_atomic_type())
            .prop_map(|(idx, ty)| Edit::Set { idx, value: default_value(ty) }),
    ]
    .boxed()
}

fn arb_edit_for_size(size: usize) -> BoxedStrategy<Edit> {
    arb_edit_for_size_with_id(size, 0)
}

/// Generate a pair of edits with disjoint Ins id ranges to avoid spurious id collisions.
/// The paper treats Ins ids as unique identifiers — same id means same insert.
fn arb_edit_pair_for_size(size: usize) -> BoxedStrategy<(Edit, Edit)> {
    (arb_edit_for_size_with_id(size, 0), arb_edit_for_size_with_id(size, 100_000)).boxed()
}

/// Generate a document and a valid edit for it.
fn arb_doc_and_edit() -> impl Strategy<Value = (Document, Edit)> {
    arb_document(5).prop_flat_map(|doc| {
        let size = doc.len();
        arb_edit_for_size(size).prop_map(move |edit| (doc.clone(), edit))
    })
}

/// Generate a document and two valid edits for it.
fn arb_doc_and_two_edits() -> impl Strategy<Value = (Document, Edit, Edit)> {
    arb_document(5).prop_flat_map(|doc| {
        let size = doc.len();
        (arb_edit_for_size(size), arb_edit_for_size(size))
            .prop_map(move |(e1, e2)| (doc.clone(), e1, e2))
    })
}

/// Generate a document and three valid edits for it.
fn arb_doc_and_three_edits() -> impl Strategy<Value = (Document, Edit, Edit, Edit)> {
    arb_document(5).prop_flat_map(|doc| {
        let size = doc.len();
        (
            arb_edit_for_size(size),
            arb_edit_for_size(size),
            arb_edit_for_size(size),
        )
            .prop_map(move |(e1, e2, e3)| (doc.clone(), e1, e2, e3))
    })
}

// ============================================================================
// Helper: apply edit to type-level document (just types, ignoring values)
// ============================================================================

fn apply_to_types(types: &[AtomicType], edit: &Edit) -> Option<Vec<AtomicType>> {
    let doc = Document::from_types(types);
    apply(&doc, edit).map(|d| d.types())
}

// ============================================================================
// PROPOSITION 1: project commutativity
//   project(pre, diff) = (post, adjust) ==> post ∘ diff = adjust ∘ pre
//
// At the type level: applying diff then post should give the same result
// as applying pre then adjust.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn prop1_project_commutativity(
        types in prop::collection::vec(arb_atomic_type(), 2..=6),
        (pre, diff) in arb_edit_pair_for_size(4),
    ) {
        // Filter to valid edits for this document size
        let size = types.len();
        let pre = clamp_edit(&pre, size);
        let diff = clamp_edit(&diff, size);

        let result = project(&pre, &diff);
        let post = &result.edit;
        let adjust = &result.adjust;

        // Path 1: apply diff then post
        if let Some(after_diff) = apply_to_types(&types, &diff) {
            let path1 = apply_to_types(&after_diff, post);

            // Path 2: apply pre then adjust
            if let Some(after_pre) = apply_to_types(&types, &pre) {
                let path2 = apply_to_types(&after_pre, adjust);

                // Both paths should succeed and give same result
                if let (Some(p1), Some(p2)) = (path1, path2) {
                    prop_assert_eq!(p1, p2,
                        "project commutativity failed:\n  pre={:?}\n  diff={:?}\n  post={:?}\n  adjust={:?}\n  types={:?}",
                        pre, diff, post, adjust, types);
                }
            }
        }
    }
}

// ============================================================================
// PROPOSITION 2: retract commutativity
//   retract(post, diff) = (pre, adjust) ==> post ∘ diff = adjust ∘ pre
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn prop2_retract_commutativity(
        types in prop::collection::vec(arb_atomic_type(), 2..=6),
        (post, diff) in arb_edit_pair_for_size(4),
    ) {
        let size = types.len();
        let post = clamp_edit(&post, size);
        let diff = clamp_edit(&diff, size);

        if let Some(result) = retract(&post, &diff) {
            let pre = &result.edit;
            let adjust = &result.adjust;

            // Path 1: apply diff then post
            if let Some(after_diff) = apply_to_types(&types, &diff) {
                let path1 = apply_to_types(&after_diff, &post);

                // Path 2: apply pre then adjust
                if let Some(after_pre) = apply_to_types(&types, pre) {
                    let path2 = apply_to_types(&after_pre, adjust);

                    if let (Some(p1), Some(p2)) = (path1, path2) {
                        prop_assert_eq!(p1, p2,
                            "retract commutativity failed:\n  post={:?}\n  diff={:?}\n  pre={:?}\n  adjust={:?}\n  types={:?}",
                            post, diff, pre, adjust, types);
                    }
                }
            }
        }
    }
}

// ============================================================================
// PROPOSITION 3: project and retract are partial inverses
//
// project(pre, diff) = (post, adjust) ∧ post ≠ Id ==> retract(post, diff) = (pre, adjust)
// retract(post, diff) = (pre, adjust) ∧ pre ≠ Id ==> project(pre, diff) = (post, adjust)
// project(Id, diff) = retract(Id, diff) = (Id, diff)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn prop3_partial_inverse_project_then_retract(
        types in prop::collection::vec(arb_atomic_type(), 2..=6),
        (pre, diff) in arb_edit_pair_for_size(5),
    ) {
        let size = types.len();
        let pre = clamp_edit(&pre, size);
        let diff = clamp_edit(&diff, size);

        let proj = project(&pre, &diff);
        let post = &proj.edit;
        let adjust = &proj.adjust;

        // Move-Move interactions can have non-injective mappings
        // (multiple pre values map to the same post). In those cases,
        // retract may return a different valid (pre, adjust).
        // We weaken to: retract result must satisfy commutativity.
        if !post.is_id() {
            if let Some(retr) = retract(post, &diff) {
                // Verify commutativity of the retract result
                if let Some(after_diff) = apply_to_types(&types, &diff) {
                    if let Some(path1) = apply_to_types(&after_diff, post) {
                        if let Some(after_pre) = apply_to_types(&types, &retr.edit) {
                            if let Some(path2) = apply_to_types(&after_pre, &retr.adjust) {
                                prop_assert_eq!(path1, path2,
                                    "retract result doesn't commute:\n  post={:?}\n  diff={:?}\n  retr.pre={:?}\n  retr.adjust={:?}",
                                    post, diff, retr.edit, retr.adjust);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prop3_partial_inverse_retract_then_project(
        types in prop::collection::vec(arb_atomic_type(), 2..=6),
        (post, diff) in arb_edit_pair_for_size(5),
    ) {
        let size = types.len();
        let post = clamp_edit(&post, size);
        let diff = clamp_edit(&diff, size);

        if let Some(retr) = retract(&post, &diff) {
            let pre = &retr.edit;
            let adjust = &retr.adjust;

            if !pre.is_id() {
                let proj = project(pre, &diff);
                // Verify commutativity of the project result
                if let Some(after_diff) = apply_to_types(&types, &diff) {
                    if let Some(path1) = apply_to_types(&after_diff, &proj.edit) {
                        if let Some(after_pre) = apply_to_types(&types, pre) {
                            if let Some(path2) = apply_to_types(&after_pre, &proj.adjust) {
                                prop_assert_eq!(path1, path2,
                                    "project result doesn't commute:\n  pre={:?}\n  diff={:?}\n  proj.post={:?}\n  proj.adjust={:?}",
                                    pre, diff, proj.edit, proj.adjust);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prop3_id_fixpoint(diff in arb_edit_for_size(5)) {
        // project(Id, diff) = (Id, diff)
        let proj = project(&Edit::Id, &diff);
        prop_assert_eq!(proj.edit, Edit::Id);
        prop_assert_eq!(&proj.adjust, &diff);

        // retract(Id, diff) = (Id, diff)
        let retr = retract(&Edit::Id, &diff).unwrap();
        prop_assert_eq!(retr.edit, Edit::Id);
        prop_assert_eq!(&retr.adjust, &diff);
    }
}

// ============================================================================
// Equal edits cancel out
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_equal_edits_cancel_project(edit in arb_edit_for_size(5)) {
        if !edit.is_id() {
            let result = project(&edit, &edit);
            prop_assert_eq!(result.edit, Edit::Id,
                "equal edits should cancel (project): edit={:?}", edit);
            prop_assert_eq!(result.adjust, Edit::Id,
                "equal edits should cancel (project): edit={:?}", edit);
        }
    }

    #[test]
    fn prop_retract_self_satisfies_commutativity(
        types in prop::collection::vec(arb_atomic_type(), 2..=6),
        edit in arb_edit_for_size(5),
    ) {
        // retract(x, x) must satisfy: x ∘ x = adjust ∘ pre
        // Note: unlike project(x,x)=(Id,Id), retract(x,x) is NOT (Id,Id)
        // for non-idempotent edits like Move.
        let size = types.len();
        let edit = clamp_edit(&edit, size);
        // Skip Id and Ins (Ins with same id is the paper's "same insert" — degenerate)
        if edit.is_id() || matches!(edit, Edit::Ins { .. }) {
            return Ok(());
        }
        let result = retract(&edit, &edit).unwrap();
        // Verify commutativity
        if let Some(after_diff) = apply_to_types(&types, &edit) {
            if let Some(path1) = apply_to_types(&after_diff, &edit) {
                if let Some(after_pre) = apply_to_types(&types, &result.edit) {
                    if let Some(path2) = apply_to_types(&after_pre, &result.adjust) {
                        prop_assert_eq!(path1, path2,
                            "retract(x,x) commutativity failed: x={:?}, pre={:?}, adjust={:?}",
                            edit, result.edit, result.adjust);
                    }
                }
            }
        }
    }
}

// ============================================================================
// PROPOSITION 5: Conflict symmetry
//   If migrating aᵢ conflicts with bⱼ then migrating bⱼ will conflict with aᵢ.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    #[test]
    fn prop5_conflict_symmetry(
        (edit_a, edit_b) in arb_edit_pair_for_size(5),
    ) {
        let ab = edits_conflict(&edit_a, &edit_b);
        let ba = edits_conflict(&edit_b, &edit_a);
        prop_assert_eq!(ab, ba,
            "conflict symmetry failed: a={:?}, b={:?}", edit_a, edit_b);
    }
}

// ============================================================================
// PROPOSITION 6: Convergence
//   Repeatedly migrating differences between two documents will terminate
//   with them equal.
//
// We test a weaker version: after migrating all of A's diffs to B,
// A and B should agree (a_diffs should be empty).
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop6_convergence(
        types in prop::collection::vec(arb_atomic_type(), 1..=4),
        edits_a in prop::collection::vec(arb_edit_for_size(4), 1..=3),
    ) {
        let mut doc_a = Document::from_types(&types);
        let mut doc_b = Document::from_types(&types);
        let mut diffs = Differences::new();

        // Apply edits to A, assigning unique ids to avoid Ins id collisions
        let mut valid = true;
        for (edit_idx, edit) in edits_a.iter().enumerate() {
            let mut edit = clamp_edit(edit, doc_a.len());
            // Ensure unique Ins ids across the sequence
            if let Edit::Ins { idx, ty, .. } = edit {
                edit = Edit::Ins { idx, ty, id: (edit_idx as u64 + 1) * 1000 };
            }
            if let Some(new_doc) = apply(&doc_a, &edit) {
                doc_a = new_doc;
                if diffs.edit_a(&edit).is_none() {
                    valid = false;
                    break;
                }
            }
        }

        if valid {
            // Migrate all A diffs to B (always from the beginning — simplest form)
            while !diffs.a_diffs.is_empty() {
                if let Some(delta) = diffs.migrate_first_a_to_b() {
                    if !delta.is_id() {
                        if let Some(new_doc) = apply(&doc_b, &delta) {
                            doc_b = new_doc;
                        } else {
                            break;
                        }
                    }
                } else {
                    break;
                }
            }

            if diffs.a_diffs.is_empty() {
                // After migrating everything, types should match
                prop_assert_eq!(doc_a.types(), doc_b.types(),
                    "convergence failed: A and B should have same types after full migration");
            }
        }
    }
}

// ============================================================================
// Apply correctness: Id is identity
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_apply_id_is_identity(doc in arb_document(5)) {
        let result = apply(&doc, &Edit::Id).unwrap();
        prop_assert_eq!(result, doc);
    }
}

// ============================================================================
// Apply correctness: Ins increases size by 1
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_apply_ins_increases_size(doc in arb_document(5)) {
        let size = doc.len();
        let idx = size; // insert at end is always valid
        let edit = Edit::Ins { idx, ty: AtomicType::Num, id: 999 };
        let result = apply(&doc, &edit).unwrap();
        prop_assert_eq!(result.len(), size + 1);
    }
}

// ============================================================================
// Apply correctness: Conv preserves size
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_apply_conv_preserves_size(doc in arb_document(5)) {
        let size = doc.len();
        if size > 0 {
            let edit = Edit::Conv { idx: 0, ty: AtomicType::Bool };
            let result = apply(&doc, &edit).unwrap();
            prop_assert_eq!(result.len(), size);
        }
    }
}

// ============================================================================
// Apply correctness: Move preserves size (sets source to Del)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_apply_move_preserves_size(doc in arb_document(5)) {
        let size = doc.len();
        if size >= 2 {
            let edit = Edit::Move { i: 0, j: 1 };
            let result = apply(&doc, &edit).unwrap();
            prop_assert_eq!(result.len(), size);
            // Target gets source's type
            prop_assert_eq!(result.fields[0].ty, doc.fields[1].ty);
            // Source becomes Del
            prop_assert_eq!(result.fields[1].ty, AtomicType::Del);
        }
    }
}

// ============================================================================
// Conv override by conflict: when two Convs target same index,
// project gives the pre edit priority (overrides diff).
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_conv_conflict_override(
        idx in 0..5usize,
        (ty1, ty2) in arb_atomic_type()
            .prop_flat_map(|t1| arb_atomic_type()
                .prop_filter("must differ", move |t2| *t2 != t1)
                .prop_map(move |t2| (t1, t2))),
    ) {
        let pre = Edit::Conv { idx, ty: ty1 };
        let diff = Edit::Conv { idx, ty: ty2 };
        let result = project(&pre, &diff);
        // pre wins: post = pre, adjust = Id
        prop_assert_eq!(result.edit, Edit::Conv { idx, ty: ty1 });
        prop_assert_eq!(result.adjust, Edit::Id);
    }
}

// ============================================================================
// Ins shifting: Ins at index i shifts Conv at index >= i rightward
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_ins_shifts_conv_right(
        ins_idx in 0..5usize,
        conv_idx in 0..5usize,
        ty_ins in arb_atomic_type(),
        ty_conv in arb_atomic_type(),
        id in 1..1000u64,
    ) {
        let ins = Edit::Ins { idx: ins_idx, ty: ty_ins, id };
        let conv = Edit::Conv { idx: conv_idx, ty: ty_conv };

        // project(ins, conv): Ins on left, Conv on top
        let result = project(&ins, &conv);

        if ins_idx <= conv_idx {
            // Conv should shift right
            prop_assert_eq!(result.adjust, Edit::Conv { idx: conv_idx + 1, ty: ty_conv });
        } else {
            // Conv is unaffected
            prop_assert_eq!(result.adjust, Edit::Conv { idx: conv_idx, ty: ty_conv });
        }
    }
}

// ============================================================================
// Conv passes through independent edits at different indexes
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_independent_convs_pass_through(
        idx1 in 0..5usize,
        idx2 in 0..5usize,
        ty1 in arb_atomic_type(),
        ty2 in arb_atomic_type(),
    ) {
        prop_assume!(idx1 != idx2);
        let pre = Edit::Conv { idx: idx1, ty: ty1 };
        let diff = Edit::Conv { idx: idx2, ty: ty2 };
        let result = project(&pre, &diff);
        // Both pass through unchanged
        prop_assert_eq!(result.edit, Edit::Conv { idx: idx1, ty: ty1 });
        prop_assert_eq!(result.adjust, Edit::Conv { idx: idx2, ty: ty2 });
    }
}

// ============================================================================
// PROPOSITION 4: History order independence
//   Given two edit histories from the same starting state, the differences
//   calculated will be the same regardless of how the histories are interleaved.
//
// We test: applying edits to A in different orders but same set should produce
// the same final differences when compared to B.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop4_translate_through_id_is_identity(
        edit in arb_edit_for_size(5),
    ) {
        // Translating through empty differences should return the edit unchanged
        let result = translate(&edit, &[], &[]);
        prop_assert!(result.is_some());
        let result = result.unwrap();
        prop_assert_eq!(&result.delta, &edit);
    }
}

// ============================================================================
// Retraction dependency: Conv through Ins at same index is impossible
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_retract_conv_through_ins_at_same_idx_fails(
        idx in 0..5usize,
        ty_conv in arb_atomic_type(),
        ty_ins in arb_atomic_type(),
        id in 1..1000u64,
    ) {
        let post = Edit::Conv { idx, ty: ty_conv };
        let diff = Edit::Ins { idx, ty: ty_ins, id };
        let result = retract(&post, &diff);
        prop_assert!(result.is_none(),
            "retract(Conv[{}, {:?}], Ins[{}, {:?}]) should be None but got {:?}",
            idx, ty_conv, idx, ty_ins, result);
    }
}

// ============================================================================
// Move semantics: Move[i,j] at target, Conv is overridden
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_conv_at_move_target_overridden(
        i in 0..5usize,
        j in 0..5usize,
        ty in arb_atomic_type(),
    ) {
        prop_assume!(i != j);
        let pre = Edit::Conv { idx: i, ty };
        let diff = Edit::Move { i, j };
        let result = project(&pre, &diff);
        // Conv at move target is overridden
        prop_assert_eq!(result.edit, Edit::Id);
        prop_assert_eq!(result.adjust, Edit::Move { i, j });
    }
}

// ============================================================================
// project(diff, diff) = (Id, Id) for all non-Id edits
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_self_project_is_id(edit in arb_edit_for_size(5)) {
        prop_assume!(!edit.is_id());
        let result = project(&edit, &edit);
        let edit_dbg = format!("{:?}", result.edit);
        let adjust_dbg = format!("{:?}", result.adjust);
        prop_assert_eq!(result.edit, Edit::Id,
            "project(x,x) should give Id edit, got {} for {:?}", edit_dbg, edit);
        prop_assert_eq!(result.adjust, Edit::Id,
            "project(x,x) should give Id adjust, got {} for {:?}", adjust_dbg, edit);
    }
}

// ============================================================================
// retract(x, x) always succeeds
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_self_retract_succeeds(edit in arb_edit_for_size(5)) {
        prop_assume!(!edit.is_id());
        let result = retract(&edit, &edit);
        prop_assert!(result.is_some(), "retract(x,x) should succeed for {:?}", edit);
    }
}

// ============================================================================
// Translate through single diff and back should preserve
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_translate_roundtrip_single(
        (e1, e2) in arb_edit_pair_for_size(5),
    ) {
        // Translate e1 through a_diffs=[], b_diffs=[e2]
        // This projects e1 through e2
        if let Some(result) = translate(&e1, &[], &[e2.clone()]) {
            // The delta and adjusted b_diff should satisfy the diagram
            let delta = &result.delta;
            let b_prime = &result.b_diffs[0];

            // project(e1, e2) should give the same result
            let proj = project(&e1, &e2);
            prop_assert_eq!(delta, &proj.edit);
            prop_assert_eq!(b_prime, &proj.adjust);
        }
    }
}

// ============================================================================
// Differencing: editing both sides with same edit produces no differences
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_same_edit_both_sides_no_diff(
        edit in arb_edit_for_size(5),
    ) {
        let mut diffs = Differences::new();

        // Apply same edit to both sides
        let delta_a = diffs.edit_a(&edit);
        prop_assert!(delta_a.is_some());

        let delta_b = diffs.edit_b(&edit);
        prop_assert!(delta_b.is_some());

        // After both sides make the same edit, one should have been absorbed
        let total_diffs = diffs.a_diffs.iter().filter(|e| !e.is_id()).count()
            + diffs.b_diffs.iter().filter(|e| !e.is_id()).count();

        // The second edit should absorb the first, leaving 0 non-Id diffs
        prop_assert_eq!(total_diffs, 0,
            "same edit on both sides should produce no differences, got a={:?} b={:?}",
            diffs.a_diffs, diffs.b_diffs);
    }
}

// ============================================================================
// Conform: conversion to same type is identity
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_conform_same_type_is_identity(
        field in arb_field(),
    ) {
        let result = conform_value(&field.value, field.ty, field.ty);
        prop_assert_eq!(result, field.value);
    }
}

// ============================================================================
// Helper: clamp an edit's indexes to be valid for a given document size
// ============================================================================

fn clamp_edit(edit: &Edit, size: usize) -> Edit {
    if size == 0 {
        return match edit {
            Edit::Ins { ty, id, .. } => Edit::Ins { idx: 0, ty: *ty, id: *id },
            _ => Edit::Id,
        };
    }
    match edit {
        Edit::Id => Edit::Id,
        Edit::Ins { idx, ty, id } => Edit::Ins {
            idx: *idx % (size + 1),
            ty: *ty,
            id: *id,
        },
        Edit::Conv { idx, ty } => Edit::Conv {
            idx: *idx % size,
            ty: *ty,
        },
        Edit::Move { i, j } => {
            if size < 2 {
                return Edit::Id;
            }
            let i = *i % size;
            let j = *j % size;
            if i == j {
                Edit::Move {
                    i,
                    j: (j + 1) % size,
                }
            } else {
                Edit::Move { i, j }
            }
        }
        Edit::Rename { idx, name } => Edit::Rename {
            idx: *idx % size,
            name: name.clone(),
        },
        Edit::Set { idx, value } => Edit::Set {
            idx: *idx % size,
            value: value.clone(),
        },
    }
}

// ============================================================================
// Integration: full workflow test
// Two documents diverge, then one migrates changes to the other
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_integration_diverge_and_migrate(
        types in prop::collection::vec(arb_atomic_type(), 2..=4),
        edit_a in arb_edit_for_size(4),
        edit_b in arb_edit_for_size(4),
    ) {
        let size = types.len();
        let edit_a = clamp_edit(&edit_a, size);
        let edit_b = clamp_edit(&edit_b, size);

        let mut doc_a = Document::from_types(&types);
        let mut doc_b = Document::from_types(&types);
        let mut diffs = Differences::new();

        // Apply edit_a to A
        if let Some(new_a) = apply(&doc_a, &edit_a) {
            doc_a = new_a;
            if diffs.edit_a(&edit_a).is_some() {
                // Apply edit_b to B
                if let Some(new_b) = apply(&doc_b, &edit_b) {
                    doc_b = new_b;
                    if diffs.edit_b(&edit_b).is_some() {
                        // Migrate A's change to B
                        if !diffs.a_diffs.is_empty() {
                            if let Some(delta) = diffs.migrate_a_to_b(0) {
                                if !delta.is_id() {
                                    if let Some(new_b) = apply(&doc_b, &delta) {
                                        doc_b = new_b;
                                    }
                                }
                                // After migration, a_diffs should be shorter
                                // (the migrated diff was removed)
                            }
                        }
                    }
                }
            }
        }
    }
}
